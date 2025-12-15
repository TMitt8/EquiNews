import torch
import argparse
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class BestOfNModel:
    def __init__(self, model_path, n=4, device="cuda", reward_model_name="maximuspowers/bias-type-classifier"):
        self.device = device
        self.n = n
        
        print(f"Loading Policy Model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        print(f"Loading Reward Model: {reward_model_name}...")
        device_id = 0 if torch.cuda.is_available() and device == "cuda" else -1
        self.bias_classifier = pipeline(
            "text-classification",
            model=reward_model_name,
            top_k=None,
            device=device_id
        )

    def compute_neutrality(self, texts):
        """
        Calculates neutrality score for a list of texts.
        Score = (1.0 - avg_top_3_probs)^2
        """
        # Process in batches to avoid OOM
        batch_size = 8
        neutrality_scores = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            # Truncate to 512 tokens for the classifier
            bias_outputs = self.bias_classifier(batch_texts, truncation=True, max_length=512)
            
            for output in bias_outputs:
                # Extract scores
                scores = sorted([float(item["score"]) for item in output], reverse=True)
                # Take top 3 (or fewer if not available)
                top3 = scores[:3] if len(scores) >= 3 else scores
                avg_top3 = np.mean(top3)
                # Calculate neutrality
                neutrality = (1.0 - avg_top3) ** 2
                neutrality_scores.append(neutrality)
                
        return np.array(neutrality_scores)

    def forward(self, input_text):
        """
        Generates the best-of-N summary for the given input text.
        Accepts a single string.
        """
        if not isinstance(input_text, str):
            raise ValueError("Input must be a single string.")
            
        # Prepare Input
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        
        # Generate N candidates
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                min_length=30,
                do_sample=True,
                top_p=0.9,
                num_beams=1,
                no_repeat_ngram_size=3,
                num_return_sequences=self.n,
                early_stopping=False
            )
        
        # Decode candidates
        candidates = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Score candidates
        rewards = self.compute_neutrality(candidates)
            
        best_idx = np.argmax(rewards)
        best_candidate = candidates[best_idx]
        best_score = rewards[best_idx]
        
        return {
            "best_summary": best_candidate,
            "best_score": best_score,
            "all_candidates": candidates,
            "all_scores": rewards.tolist()
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Best-of-N inference for unbiased summarization.")
    parser.add_argument("--input_file", type=str, help="Path to a text file containing the input text to summarize.")
    args = parser.parse_args()

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sft_model_path = "./sft_summarizer_final" # Path to your fine-tuned model
    
    try:
        # Initialize the wrapper class
        model = BestOfNModel(sft_model_path, n=4, device=device)
    except OSError:
        print(f"Could not find model at {sft_model_path}. Please ensure the path is correct or run the SFT training first.")
        return

    # Determine Input Text
    if args.input_file:
        if os.path.exists(args.input_file):
            with open(args.input_file, "r", encoding="utf-8") as f:
                input_text = f.read()
            print(f"Loaded input from {args.input_file}")
        else:
            print(f"Error: File {args.input_file} not found.")
            return
    else:
        # Example Input
        input_text = """
        The controversial bill was passed yesterday amid fierce protests. Critics argue it undermines democracy, while supporters claim it is necessary for national security. The opposition leader called it a "dark day," whereas the Prime Minister hailed it as a "historic victory."
        """
    
    print("\nInput Text:")
    print(input_text.strip())
    print("-" * 50)
    
    # Run Inference
    print(f"Generating 4 candidates...")
    print("Scoring candidates...")
    result = model.forward(input_text)
    
    print(f"\n--- Best Selection (Score: {result['best_score']:.4f}) ---")
    print(result['best_summary'])
    print("\n--- All Candidates ---")
    for i, (cand, score) in enumerate(zip(result['all_candidates'], result['all_scores'])):
        is_best = "*" if i == result['all_scores'].index(result['best_score']) else " "
        print(f"[{i+1}] {is_best} Score: {score:.4f} | {cand}")

if __name__ == "__main__":
    main()
