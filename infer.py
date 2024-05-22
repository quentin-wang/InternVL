import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer
from glob import glob

imgdir = r"/mnt/workspace/wbin/datasets/reading"
imglist = glob(imgdir + "/*.jpg")

path = "/mnt/workspace/wbin/modelhub/InternVL-Chat-V1-5"
# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()

print(model)

# Otherwise, you need to set device_map='auto' to use multiple GPUs for inference.
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     device_map='auto').eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

for imgname in imglist:
    image = Image.open(imgname).convert('RGB')
    # image = Image.open('./examples/image2.jpg').convert('RGB')
    image = image.resize((448, 448))
    image_processor = CLIPImageProcessor.from_pretrained(path)

    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    generation_config = dict(
        num_beams=1,
        max_new_tokens=512,
        do_sample=False,
    )

    # single-round conversation
    question = "请详细描述图片"
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(question, response)
