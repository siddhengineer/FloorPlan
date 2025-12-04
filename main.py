"""
Main entry point for FloorPlan Vision AI testing.
"""

from floorplan import FloorPlanVisionAI


def main():
    """Main function to test the FloorPlan Vision AI model."""
    
    # Initialize the model
    print("Initializing FloorPlan Vision AI model...")
    model = FloorPlanVisionAI()
    
    # Example: Test with a local image
    image_path = "floor_plan.png"  # Update with your image path
    
    # Different prompts to test
    prompts = [
        "Describe this floor plan in detail.",
        "What are the main rooms in this floor plan?",
        "What is the total square footage?",
        "Analyze the layout and flow of this floor plan.",
        "List all the rooms and their approximate dimensions."
    ]
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {image_path}")
    print(f"{'='*60}\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Query {i} ---")
        print(f"Prompt: {prompt}")
        print("\nResponse:")
        print("-" * 60)
        
        try:
            result = model.analyze_floor_plan(image_path, prompt)
            print(result)
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found.")
            print("Please update the image_path variable with your floor plan image.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("-" * 60)


if __name__ == "__main__":
    main()
