import torch
import transformers
import coremltools as ct
import numpy as np


# For bug filing
print(
    "Versions:\n"
    f"torch: {torch.__version__}\n"
    f"transformers: {transformers.__version__}\n"
    f"coremltools: {ct.__version__}\n"
    f"numpy: {np.__version__}"
)

model = (
    transformers.AutoModelForSequenceClassification.from_pretrained(
        "yikuan8/Clinical-Longformer",
        num_labels=1,
        problem_type="regression",
        torch_dtype=torch.float32,
        torchscript=True,
        return_dict=False,  # Need this, or forward will bork
    )
    .to(device="mps", dtype=torch.float32)
    .eval()
)

# skip as_strided, unfold, etc. by pretending we're exporting to ONNX
# see https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py#L780
model.config.onnx_export = True

# mimic batched inputs
batched_input_shape = (4, 4096)
fake_inputs = torch.randint(50_000, batched_input_shape).to(device="mps")
# No "wasted tokens" so pay attention to everything
fake_attn_mask = torch.ones_like(fake_inputs).to(dtype=int, device="mps")

fake_input_dict = {"input_ids": fake_inputs, "attention_mask": fake_attn_mask}

traced = torch.jit.trace(model, example_kwarg_inputs=fake_input_dict)

# Convenience aliases
register_torch_op = ct.converters.mil.frontend.torch.register_torch_op
_get_inputs = ct.converters.mil.frontend.torch.ops._get_inputs
_make_fill_op = ct.converters.mil.frontend.torch.ops._make_fill_op


@register_torch_op
def new_ones(context, node):
    inputs = _get_inputs(context, node)
    size = inputs[1]
    result = _make_fill_op(size, 1.0, node.name)
    context.add(result)


# Can replace one of the shape elements with a ct.RangeDim(lower, upper) for variable size stuff.
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=batched_input_shape, dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=batched_input_shape, dtype=np.int32),
    ],
    minimum_deployment_target=ct.target.macOS14,  # Alias for iOS17 target
)
