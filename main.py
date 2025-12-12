from ppo_trainer import AgentLLM
import torch
import yaml
from types import SimpleNamespace
from datetime import datetime
import os
from pdf_generator import PDFGenerator


def main(args):

    agent_llm = AgentLLM(args)
    
    # Create output file for questions and answers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"generated_questions_answers_{timestamp}.pdf"
    
    # Initialize PDF generator
    pdf_gen = PDFGenerator(output_file)
    
    pdf_gen.add_header("GENERATED QUESTIONS AND ANSWERS")
    pdf_gen.add_text(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf_gen.add_text(f"Total iterations: {args.num_iterations}")
    pdf_gen.add_separator()
    
    print(f"📝 Saving questions and answers to: {output_file}")

    # Loop to iteratively refine responses using PPO
    for iteration in range(args.num_iterations):  # Number of iterations specified in config
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}/{args.num_iterations}")
        print(f"{'='*80}\n")
        
        try:
            # Run PPO step - questions are generated internally from random walks
            decoded_responses, rewards, stats, batch_meta = agent_llm.ppo_step()
            
            # Add iteration header to PDF
            pdf_gen.add_header(f"ITERATION {iteration + 1}", level=2)
            pdf_gen.add_text("Generated Responses:")
            pdf_gen.add_separator()
            
            for idx, response in enumerate(decoded_responses):
                # Get walk and hidden node information from batch_meta
                meta = batch_meta[idx] if idx < len(batch_meta) else {}
                # walk = meta.get("walk", []) # Removed from batch_meta in previous step
                hidden_node = meta.get("hidden_node", "N/A")
                visible_nodes = meta.get("visible_nodes", [])
                
                pdf_gen.add_text(f"[Response {idx + 1}]")
                # pdf_gen.add_text(f"Random walk: {walk}")
                pdf_gen.add_text(f"Hidden node: {hidden_node}")
                pdf_gen.add_text(f"Visible nodes: {visible_nodes}")
                pdf_gen.add_text(f"Reward: {rewards[idx] if idx < len(rewards) else 'N/A'}")
                pdf_gen.add_separator()
                
                # Split into question and answer if "Answer:" is present
                if "Answer:" in response:
                    question_part, answer_part = response.split("Answer:", 1)
                    pdf_gen.add_text("Question:")
                    pdf_gen.add_text_with_latex(question_part.strip())
                    pdf_gen.add_text("Answer:")
                    pdf_gen.add_text_with_latex(answer_part.strip())
                else:
                    pdf_gen.add_text("Full Response:")
                    pdf_gen.add_text_with_latex(response.strip())
                
                pdf_gen.add_separator()
            
        except Exception as e:
            print(f"❌ Error during PPO step: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next iteration instead of stopping
            continue
    
    # Write summary at the end
    pdf_gen.add_separator()
    pdf_gen.add_header("TRAINING COMPLETED", level=2)
    pdf_gen.add_text(f"Training ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save the PDF
    try:
        pdf_gen.save()
        print(f"\n✅ All questions and answers saved to: {output_file}")
    except Exception as e:
        print(f"\n❌ Error saving PDF: {e}")


if __name__ == "__main__":
    args = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
    args = SimpleNamespace(**args)
    main(args)
