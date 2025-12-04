"""
FloorPlanVisionAIAdaptor Model Implementation and Testing
Model: https://huggingface.co/sabaridsnfuji/FloorPlanVisionAIAdaptor
"""

from unsloth import FastVisionModel
from PIL import Image
import torch
from transformers import TextStreamer
import os


class FloorPlanVisionAI:
    """Wrapper class for FloorPlanVisionAIAdaptor model."""
    
    def __init__(self, model_name: str = "sabaridsnfuji/FloorPlanVisionAIAdaptor"):
        """
        Initialize the FloorPlanVisionAI model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if self.device == "cpu":
            print("WARNING: This model requires CUDA/GPU. CPU inference may not work properly.")
        
        # Load the pre-trained model and tokenizer using unsloth
        print("Loading model with FastVisionModel...")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=True,  # Use 4-bit precision to save memory
            use_gradient_checkpointing="unsloth"  # Enable gradient checkpointing for efficiency
        )
        
        # Enable inference mode
        FastVisionModel.for_inference(self.model)
        
        print("Model loaded successfully!")
    
    def analyze_floor_plan(
        self, 
        image_path: str, 
        instruction: str = "You are an expert in architecture and interior design. Analyze the floor plan image and describe accurately the key features, room count, layout, and any other important details you observe.",
        max_new_tokens: int = 2048,
        temperature: float = 1.5,
        min_p: float = 0.1,
        stream_output: bool = True
    ):
        """
        Analyze a floor plan image.
        
        Args:
            image_path: Path to local floor plan image
            instruction: Text instruction for the model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            min_p: Minimum probability threshold
            stream_output: Whether to stream output to console
            
        Returns:
            Generated text description
        """
        # Load image
        try:
            image = Image.open(image_path)
            print(f"Image loaded: {image_path}")
            print(f"Image size: {image.size}")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        
        # Format input message
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        
        # Prepare inputs
        inputs = self.tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        
        # Setup text streamer if streaming is enabled
        text_streamer = TextStreamer(self.tokenizer, skip_prompt=True) if stream_output else None
        
        # Generate response
        print("\nGenerating response...\n")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                streamer=text_streamer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                min_p=min_p
            )
        
        # Decode and return
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


def test_with_local_image(image_path: str):
    """
    Test the model with a local floor plan image.
    
    Args:
        image_path: Path to local floor plan image
    """
    print("="*60)
    print("Testing FloorPlanVisionAIAdaptor with Unsloth")
    print("="*60)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Initialize model
    model = FloorPlanVisionAI()
    
    # Default instruction
    instruction = """You are an expert in architecture and interior design. Analyze the floor plan image and describe accurately the key features, room count, layout, and any other important details you observe."""
    
    print(f"\nImage: {image_path}")
    print(f"Instruction: {instruction}")
    print("\n" + "="*60)
    print("Response:")
    print("="*60 + "\n")
    
    try:
        result = model.analyze_floor_plan(
            image_path, 
            instruction=instruction,
            stream_output=True  # Stream output to console
        )
        
        print("\n" + "="*60)
        print("Analysis complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


def test_multiple_prompts(image_path: str):
    """
    Test the model with multiple prompts on the same image.
    
    Args:
        image_path: Path to local floor plan image
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    print("="*60)
    print("Testing Multiple Prompts")
    print("="*60)
    
    # Initialize model once
    model = FloorPlanVisionAI()
    
    # Different analysis prompts
    prompts = [
        "What are the dimensions and total square footage of this floor plan?",
        "List all the rooms in this floor plan and their purposes.",
        "Describe the flow and circulation patterns in this layout.",
        "What are the strengths and weaknesses of this floor plan design?",
        "Suggest improvements for this floor plan layout."
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"Prompt {i}/{len(prompts)}")
        print(f"{'='*60}")
        print(f"{prompt}\n")
        
        try:
            result = model.analyze_floor_plan(
                image_path,
                instruction=prompt,
                stream_output=True
            )
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print()


if __name__ == "__main__":
    import sys
    
    print("FloorPlanVisionAIAdaptor Test Suite (Unsloth)\n")
    
    # Check if image path provided as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Check for --multi flag
        if len(sys.argv) > 2 and sys.argv[2] == "--multi":
            test_multiple_prompts(image_path)
        else:
            test_with_local_image(image_path)
    else:
        print("No image path provided.")
        print("\nUsage:")
        print("  python floorplan.py <path_to_floor_plan_image>")
        print("  python floorplan.py <path_to_floor_plan_image> --multi")
        print("\nExamples:")
        print("  python floorplan.py floor_plan.png")
        print("  python floorplan.py floor_plan.png --multi")
        print("\nThe --multi flag tests multiple different prompts on the same image.")