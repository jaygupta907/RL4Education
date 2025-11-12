from reward_model import RewardModel
import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM
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

        # Load the base model with value head
        base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map="auto"
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
        base_model.pretrained_model = get_peft_model(
            base_model.pretrained_model, lora_config
        )

        # Set the model as the LoRA-adapted model
        self.model = base_model

        # Load the reference model (frozen copy)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # PPO configuration
        self.ppo_config = PPOConfig(
            model_name=self.model_name,
            learning_rate=1.41e-5,
            batch_size=8,
            mini_batch_size=2,
            gradient_accumulation_steps=1,
            optimize_cuda_cache=True,
            target_kl=0.1,
            ppo_epochs=4,
            seed=0,
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

    def generate_query(self, question):
        extracted_concepts = self.reward_model.extract_concepts(question)
        print("-" * shutil.get_terminal_size().columns)
        print(f"Extracted concepts for the question: {extracted_concepts}")
        print("-" * shutil.get_terminal_size().columns)
        dependent_concepts = self.concept_graph.get_dependents(extracted_concepts)
        print("-" * shutil.get_terminal_size().columns)
        print(f"New concepts for the question: {dependent_concepts}")
        print("-" * shutil.get_terminal_size().columns)
        context = []
        for concept in dependent_concepts:
            context_piece = self.context_loader.get_context(concept)
            context.append(context_piece)
        batch_prompts = []
        batch_meta = []  # keep track of per-prompt metadata
        for new_concept, context_piece in zip(dependent_concepts, context):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert at creating challenging educational questions. Always output the question followed by its answer.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Original question: {question}\n\n"
                            f"This question covers: {', '.join(extracted_concepts)}\n\n"
                            f"Task: Create a harder question that also incorporates the concept '{new_concept}'.\n\n"
                            f"Context about '{new_concept}':\n{context_piece}\n\n"
                            f"Requirements:\n"
                            f"- Make it more challenging than the original\n"
                            f"- Integrate '{new_concept}' naturally\n"
                            f"- Ensure the question remains answerable strictly from the provided context\n"
                            f"- Output the question and answer\n\n"
                            f"- Make sure the answer part starts with 'Answer'\n"
                        ),
                    },
                ]

                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except:
                # # Fallback to manual formatting if chat template fails
                # prompt = (
                #     f"<|im_start|>system\n"
                #     f"You are an expert question generator. Output the question followed by its answer.<|im_end|>\n"
                #     f"<|im_start|>user\n"
                #     f"Original question: {question}\n"
                #     f"Concepts covered: {', '.join(extracted_concepts)}\n\n"
                #     f"Create a harder question that includes '{new_concept}'.\n"
                #     f"Context: {context_piece}\n\n"
                #     f"Output the question followed by its answer:<|im_end|>\n"
                #     f"<|im_start|>assistant\n"
                # )

                # prompt = self.tokenizer.apply_chat_template(
                #     messages,
                #     tokenize=False,
                #     add_generation_prompt=True
                # )
                pass
            batch_prompts.append(prompt)
            allowed_concepts = set(extracted_concepts) | {new_concept}
            batch_meta.append(
                {
                    "new_concept": new_concept,
                    "context": context_piece,
                    "allowed_concepts": list(allowed_concepts),
                    "extracted_concepts": list(extracted_concepts),
                }
            )

        target_size = self.ppo_config.batch_size

        if len(batch_prompts) < target_size:
            import random

            while len(batch_prompts) < target_size:
                idx = random.randrange(len(batch_prompts))
                batch_prompts.append(batch_prompts[idx])
                batch_meta.append(batch_meta[idx])
        elif len(batch_prompts) > target_size:
            import random

            indices = list(range(len(batch_prompts)))
            indices = random.sample(indices, target_size)
            batch_prompts = [batch_prompts[i] for i in indices]
            batch_meta = [batch_meta[i] for i in indices]

        return batch_prompts, batch_meta

    def ppo_step(self, question):
        batch_prompts, batch_meta = self.generate_query(question)

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
            max_new_tokens=150,
            min_new_tokens=10,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
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

            # Pass only the question to the reward model
            if len(question_part.strip()) == 0:
                # Penalize empty questions
                final_score = -1.0
            else:
                allowed_concepts = batch_meta[idx].get("allowed_concepts", [])
                base_score = self.reward_model.get_reward(
                    question_part.strip(),
                    allowed_concepts=allowed_concepts,
                    answer=answer_part.strip(),
                )
                word_count = len(question_part.split())
                length_bonus = min(word_count / 20.0, 1.0) * 0.5

                final_score = base_score + length_bonus

            wandb.log(
                {
                    "response": response,
                    "reward": final_score,
                    "question": question_part,
                    "answer": answer_part,
                }
            )
            rewards.append(torch.tensor(final_score).to(self.device))
            print("=" * shutil.get_terminal_size().columns)
            print(
                f"Question: '{question_part.strip()}'\n"
                f"Generated Answer: '{answer_part.strip()}'\n"
                f"Reward: {final_score:.3f}"
            )
            print("=" * shutil.get_terminal_size().columns)

        # PPO step expects query_tensors and response_tensors separately
        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
        torch.cuda.empty_cache()

        return decoded_responses, rewards, stats
