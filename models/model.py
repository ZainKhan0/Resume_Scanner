from transformers import AutoModel, AutoTokenizer
import torch

#Loading the pre-trained Bert and tokenizer.....   
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def encode_text(text):
    """Tokenize and encode text using BERT model"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def match_resume_to_job(resume_text, job_description):
    """Compute similarity between resume and the JD"""
    resume_embedding = encode_text(resume_text)
    job_embedding = encode_text(job_description)


    similarity = torch.nn.functional.cosine_similarity(resume_embedding, job_embedding)
    return similarity.item()