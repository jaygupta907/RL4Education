from reward_model import RewardModel
import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from load_context import LoadContext
from concept_graph import ConceptGraph
import random
import shutil
import wandb


class AgentLLM:
    def __init__(self, args):
        self.model_name = args.model_name
        self.concept_graph_path = args.concept_graph_path
        self.question_bank_path = args.question_bank_path
        self.reward_model = RewardModel(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the base model with value head - use 4-bit quantization to save memory
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            quantization_config=quantization_config
        )

        wandb.init(
            entity="jay_gupta-indian-institute-of-technology-madras",
            project="rl4edu-ppo-lora",
            name="ppo-lora-run",
        )

        # Prepare the model for LoRA fine-tuning
        base_model.pretrained_model = prepare_model_for_kbit_training(
            base_model.pretrained_model
        )

        # Configure LoRA
        lora_config = LoraConfig(
            r=8,  # Rank of the LoRA matrices
            lora_alpha=32,  # Scaling factor
            target_modules=["q_proj", "v_proj"],  # Target attention layers
            lora_dropout=0.1,  # Dropout for LoRA layers
            bias="none",  # No bias adaptation
            task_type="CAUSAL_LM",  # Task type: causal language modeling
        )

        # Apply LoRA to the base model's pretrained model
        lora_config = LoraConfig(
            r=8,  # Rank of the LoRA matrices
            lora_alpha=32,  # Scaling factor
            target_modules=["q_proj", "v_proj"],  # Target attention layers
            lora_dropout=0.1,  # Dropout for LoRA layers
            bias="none",  # No bias adaptation
            task_type="CAUSAL_LM",  # Task type: causal language modeling
        )
        
        base_model.pretrained_model = get_peft_model(
            base_model.pretrained_model, lora_config
        )
        
        # Enable gradient checkpointing to save memory
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False

        # Set the model as the LoRA-adapted model
        self.model = base_model

        # Load the reference model (frozen copy) - use 4-bit quantization to save memory
        # Note: PPO requires ref_model on same device as main model during training
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            quantization_config=quantization_config
        )
        
        # Check if the model is DeepSeek and set pad_token_id if needed
        # DeepSeek models sometimes have missing pad_token in config
        if "deepseek" in self.model_name.lower():
             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
             if self.tokenizer.pad_token is None:
                 self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
             # Load the tokenizer
             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
             self.tokenizer.pad_token = self.tokenizer.eos_token

        # PPO configuration - minimal batch size to save memory
        # Note: batch_size must be divisible by (mini_batch_size * gradient_accumulation_steps)
        self.ppo_config = PPOConfig(
            model_name=self.model_name,
            learning_rate=5e-6,  # Reduced learning rate to prevent divergence
            batch_size=1,  # Minimal batch size to save memory (ref_model must stay on GPU)
            mini_batch_size=1,  # Minimal to save memory
            gradient_accumulation_steps=1,  # batch_size (1) must be divisible by mini_batch_size (1) * gradient_accumulation_steps (1) = 1
            optimize_cuda_cache=True,
            gradient_checkpointing=True,
            target_kl=0.2,  # Increased target KL to allow more exploration while controlling divergence
            init_kl_coef=1.0,  # Increased KL penalty coefficient to strongly prevent divergence
            kl_penalty="kl",  # Use KL penalty to control divergence
            ppo_epochs=2,  # Reduced from 4 to save memory
            seed=0,
            cliprange=0.2,  # Slightly tighter clipping range for stability
            cliprange_value=0.2,  # Value function clipping
            vf_coef=0.5,  # Value function coefficient
        )

        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

        # Load the concept graph and context loader
        self.concept_graph = ConceptGraph(self.concept_graph_path)
        self.context_loader = LoadContext(self.question_bank_path)
        # Random walk parameters
        self.walk_length = getattr(args, "walk_length", 5)

    def _create_prompt(self, visible_nodes, visible_contexts, hidden_node, hidden_context):
        """Helper function to create a prompt from visible nodes and hidden node."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert at creating challenging educational questions that test "
                        "deep understanding and integrate multiple concepts. Always provide the question "
                        "followed by a comprehensive answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Create a question based on these concepts: {', '.join(visible_nodes)}\n\n"
                        f"Context for these concepts:\n" + "\n\n".join(visible_contexts) + "\n\n"
                        f"Target topic (the answer should relate to this): {hidden_node}\n"
                        f"Context for target topic:\n{hidden_context}\n\n"
                        f"Requirements:\n"
                        f"- Create a question that can be answered using the provided context\n"
                        f"- Integrate multiple concepts from the list\n"
                        f"- The question should be aimed at answering or covering the topic: {hidden_node}\n"
                        f"- The answer should relate to and explain {hidden_node} based on its context above\n"
                        f"- Test deep conceptual understanding\n"
                        f"- Make the question challenging and thought-provoking\n"
                        f"- The answer should demonstrate understanding of how these concepts connect to {hidden_node}\n\n"
                        f"Format: Question first, then 'Answer:' followed by a comprehensive explanation."
                    ),
                },
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback to manual formatting if chat template fails
            prompt = (
                f"<|im_start|>system\n"
                f"You are an expert at creating challenging educational questions that test "
                f"deep understanding and integrate multiple concepts. Always provide the question "
                f"followed by a comprehensive answer.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Create a question based on these concepts: {', '.join(visible_nodes)}\n\n"
                f"Context for these concepts:\n" + "\n\n".join(visible_contexts) + "\n\n"
                f"Target topic (the answer should relate to this): {hidden_node}\n"
                f"Context for target topic:\n{hidden_context}\n\n"
                f"Requirements:\n"
                f"- Create a question that can be answered using the provided context\n"
                f"- Integrate multiple concepts from the list\n"
                f"- The question should be aimed at answering or covering the topic: {hidden_node}\n"
                f"- The answer should relate to and explain {hidden_node} based on its context above\n"
                f"- Test deep conceptual understanding\n"
                f"- Make the question challenging and thought-provoking\n"
                f"- The answer should demonstrate understanding of how these concepts connect to {hidden_node}\n\n"
                f"Format: Question first, then 'Answer:' followed by a comprehensive explanation.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        return prompt

    def _generate_prompt_from_walk(self, walk):
        """
        Helper function to generate a prompt and metadata from a random walk.
        
        Args:
            walk: List of concepts representing a random walk
            
        Returns:
            tuple: (prompt, meta_dict) or None if walk is invalid
        """
        # Validate walk length
        if len(walk) < 3:
            return None
        
        # Always hide the last node in the random walk
        hidden_node = walk[-1]  # Last node in the walk
        visible_nodes = walk[:-1]  # All nodes except the last one
        
        if not visible_nodes:
            return None
        
        # Get context for visible nodes
        visible_contexts = []
        for node in visible_nodes:
            context_piece = self.context_loader.get_context(node)
            visible_contexts.append(f"{node}: {context_piece}")
        
        # Get context for hidden node (for verification)
        hidden_context = self.context_loader.get_context(hidden_node)
        
        prompt = self._create_prompt(visible_nodes, visible_contexts, hidden_node, hidden_context)
        meta = {
            "hidden_node": hidden_node,
            "hidden_context": hidden_context,
            "visible_nodes": visible_nodes,
            "walk": walk,
        }
        
        return prompt, meta
    
    def generate_query(self):
        batch_prompts = []
        batch_meta = []  # keep track of per-prompt metadata
        
        target_size = self.ppo_config.batch_size
        max_total_attempts = 100  # Maximum total attempts to get valid prompts
        
        print("-" * shutil.get_terminal_size().columns)
        print(f"Generating {target_size} batch items...")
        print("-" * shutil.get_terminal_size().columns)
        
        attempts = 0
        while len(batch_prompts) < target_size and attempts < max_total_attempts:
            attempts += 1
            
            # Generate a new walk for each batch item to ensure diversity
            walk = self.concept_graph.random_walk(self.walk_length)
            
            # Retry if walk is too short
            retry_count = 0
            while len(walk) < 3 and retry_count < 10:
                walk = self.concept_graph.random_walk(self.walk_length)
                retry_count += 1
            
            # Generate prompt from walk
            result = self._generate_prompt_from_walk(walk)
            if result is None:
                continue
            
            prompt, meta = result
            
            # Print the walk and corresponding hidden/visible nodes for this batch item
            print(f"\nBatch item {len(batch_prompts) + 1}:")
            print(f"  Random walk: {meta['walk']}")
            print(f"  Hidden node: {meta['hidden_node']}")
            print(f"  Visible nodes: {meta['visible_nodes']}")
            
            batch_prompts.append(prompt)
            batch_meta.append(meta)
        
        # Warn if we couldn't generate enough prompts
        if len(batch_prompts) < target_size:
            print(f"⚠️  WARNING: Only generated {len(batch_prompts)}/{target_size} prompts. Continuing with available prompts.")
        
        # If we have more than needed, randomly sample
        if len(batch_prompts) > target_size:
            indices = list(range(len(batch_prompts)))
            indices = random.sample(indices, target_size)
            batch_prompts = [batch_prompts[i] for i in indices]
            batch_meta = [batch_meta[i] for i in indices]

        return batch_prompts, batch_meta

    def ppo_step(self):
        batch_prompts, batch_meta = self.generate_query()

        # Tokenize with proper padding
        tokenized = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        ).to(self.device)

        # Extract query tensors as list
        query_tensors = [ids for ids in tokenized.input_ids]

        generation_kwargs = dict(
            max_new_tokens=1024,
            min_new_tokens=10,
            do_sample=True,
            top_k=50,
            top_p=0.9,  # Slightly lower top_p for more focused generation
            temperature=0.7,  # Lower temperature for more stable, less diverse generation
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            return_prompt=False,  # Explicitly return only generated tokens
        )

        # Generate responses
        response_tensors = self.ppo_trainer.generate(query_tensors, **generation_kwargs)

        # Decode responses - response_tensors are ONLY the generated tokens
        decoded_responses = []
        for i, response_ids in enumerate(response_tensors):
            # Remove padding tokens from response
            response_ids = response_ids[response_ids != self.tokenizer.pad_token_id]

            decoded_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # DeepSeek-R1 specific cleaning
            if "deepseek" in self.model_name.lower():
                # Remove <think>...</think> tags if present
                if "<think>" in decoded_text:
                     # We might want to keep the thought process or remove it.
                     # For now, let's remove it to just get the final answer which usually follows
                     # Or if the model output format is different, we might need to adjust.
                     # Often DeepSeek-R1 outputs: <think>\n...\n</think>\nAnswer...
                     parts = decoded_text.split("</think>")
                     if len(parts) > 1:
                         decoded_text = parts[-1].strip()
            
            decoded_responses.append(decoded_text)

        # Calculate rewards
        rewards = []
        for idx, response in enumerate(decoded_responses):
            # Split the response into question and answer
            if "Answer:" in response:
                question_part, answer_part = response.split("Answer:", 1)
            else:
                question_part = response
                answer_part = ""

            # Calculate rewards - only verification reward
            verification_score = 0.0
            if len(question_part.strip()) == 0:
                # Penalize empty questions
                final_score = -1.0
            else:
                # Verification: Check if answer relates to answers in hidden node context
                hidden_node = batch_meta[idx].get("hidden_node", "")
                hidden_context = batch_meta[idx].get("hidden_context", "")
                
                if hidden_node and hidden_context and len(answer_part.strip()) > 0:
                    try:
                        # Extract answers from hidden context (format: "Retrieved Question: ...\nRetrieved Answer: ...")
                        hidden_answers = []
                        for line in hidden_context.split("\n\n"):
                            if "Retrieved Answer:" in line:
                                # Extract the answer part after "Retrieved Answer:"
                                answer_text = line.split("Retrieved Answer:", 1)[1].strip()
                                if answer_text:
                                    hidden_answers.append(answer_text)
                        
                        if hidden_answers:
                            # Compare generated answer against all answers from hidden node
                            emb = self.reward_model.embedding_model
                            # Batch encode for efficiency
                            all_texts = [answer_part.strip()] + hidden_answers
                            all_embs = emb.encode(all_texts, convert_to_tensor=True, show_progress_bar=False)
                            
                            answer_emb = all_embs[0:1]
                            hidden_answer_embs = all_embs[1:]
                            
                            # Compute similarity with each hidden answer and take maximum
                            similarities = torch.cosine_similarity(
                                answer_emb.unsqueeze(0), 
                                hidden_answer_embs
                            )
                            max_similarity = similarities.max().item()
                            
                            # Map cosine similarity [-1,1] to [0,1] for verification score
                            verification_score = max(0.0, min((max_similarity + 1.0) / 2.0, 1.0))
                            
                            # Clear GPU memory
                            del all_embs, answer_emb, hidden_answer_embs, similarities
                            torch.cuda.empty_cache()
                        else:
                            # Fallback: compare against full context if no answers extracted
                            emb = self.reward_model.embedding_model
                            all_texts = [answer_part.strip(), hidden_context]
                            all_embs = emb.encode(all_texts, convert_to_tensor=True, show_progress_bar=False)
                            similarity = torch.cosine_similarity(all_embs[0:1], all_embs[1:2]).item()
                            verification_score = max(0.0, min((similarity + 1.0) / 2.0, 1.0))
                            
                            # Clear GPU memory
                            del all_embs
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Verification error: {e}")
                        verification_score = 0.0
                
                # Final score: only verification reward
                final_score = verification_score

            hidden_node = batch_meta[idx].get("hidden_node", "N/A")
            wandb.log(
                {
                    "response": response,
                    "reward": final_score,
                    "question": question_part,
                    "answer": answer_part,
                    "hidden_node": hidden_node,
                    "verification_score": verification_score,
                }
            )
            rewards.append(torch.tensor(final_score).to(self.device))
            print("=" * shutil.get_terminal_size().columns)
            print(
                f"Question: '{question_part.strip()}'\n"
                f"Generated Answer: '{answer_part.strip()}'\n"
                f"Hidden Node: {hidden_node}\n"
                f"Verification Score: {verification_score:.3f}\n"
                f"Reward: {final_score:.3f}"
            )
            print("=" * shutil.get_terminal_size().columns)

        # Clear memory before PPO step
        torch.cuda.empty_cache()
        
        # PPO step expects query_tensors and response_tensors separately
        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Aggressive memory cleanup after PPO step
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return decoded_responses, rewards, stats, batch_meta
