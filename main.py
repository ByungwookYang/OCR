from pdf2image import convert_from_path
import pytesseract
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch, json
 
# 1) PDF -> 이미지 변환
images = convert_from_path("sample_invoice.pdf")
image = images[0].convert("RGB")
 
# 2) OCR 추출
ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
words, boxes = [], []
for i in range(len(ocr_data["text"])):
    if ocr_data["text"][i].strip():
        words.append(ocr_data["text"][i])
        (x, y, w, h) = (ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i])
        boxes.append([x, y, x + w, y + h])
 
# 3) Processor 준비
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-large")
 
# 4) 인코딩
encoding = processor(image, words, boxes=boxes, return_tensors="pt", truncation=True)
 
# 5) 모델 로딩
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-large",
    num_labels=10  # 예: Key/Value/Table 등 라벨 수 맞게 설정
)
 
# 6) 추론
with torch.no_grad():
    outputs = model(**encoding)
preds = torch.argmax(outputs.logits, dim=-1)
 
# 7) JSON 변환
results = []
for word, box, pred in zip(words, boxes, preds[0].tolist()):
    results.append({
        "text": word,
        "bbox": box,
        "label": int(pred)
    })
 
with open("layoutlmv3_large_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
 
print("JSON 저장 완료: layoutlmv3_large_result.json")
