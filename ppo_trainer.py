from reward_model import RewardModel
import torch
from trl import PPOTrainer, PPOConfig,AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from load_context import LoadContext
from concept_graph import ConceptGraph



class AgentLLM:
    def __init__(self, args):
        self.model_name = args.model_name
        self.concept_graph_path = args.concept_graph_path
        self.question_bank_path = args.question_bank_path
        self.reward_model = RewardModel(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )

        # 3. Load the reference model (frozen copy)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )

        # 4. Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
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

        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer
        )
        self.concept_graph = ConceptGraph(self.concept_graph_path)
        self.context_loader = LoadContext(self.question_bank_path)

    def generate_query(self,question):
        extracted_concepts = self.reward_model.extract_concepts(question)
        dependent_concepts = self.concept_graph.get_dependents(extracted_concepts)
        context = []
        for concept in dependent_concepts:
            context_piece= self.context_loader.get_context(concept)
            context.append(context_piece)
        batch_prompts = []
        for new_concept, context_piece in zip(dependent_concepts, context):
            prompt = (
                f"Given the question: '{question}', which involves {', '.join(extracted_concepts)}, "
                f"generate a harder question that also includes the concept '{new_concept}'. "
                f"Here is some context about '{new_concept}': {context_piece}. Just provide the new question.New Question:"
            )
            batch_prompts.append(prompt)
        
        return batch_prompts
    
    def ppo_step(self, question):
        batch_prompts = self.generate_query(question)
        
        # Tokenize with proper padding
        tokenized = self.tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            padding_side='left'
        ).to(self.device)
        
        # Extract query tensors as list
        query_tensors = [ids for ids in tokenized.input_ids]
        
        generation_kwargs = dict(
            max_new_tokens=100,
            min_new_tokens=10,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            return_prompt=False  # Explicitly return only generated tokens
        )
        
        # Generate responses
        response_tensors = self.ppo_trainer.generate(
            query_tensors,
            **generation_kwargs
        )
        
        # Decode responses - response_tensors are ONLY the generated tokens
        decoded_responses = []
        for i, response_ids in enumerate(response_tensors):
            # Remove padding tokens from response
            response_ids = response_ids[response_ids != self.tokenizer.pad_token_id]
            
            decoded_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            decoded_responses.append(decoded_text)
        
        # Calculate rewards
        rewards = []
        for response in decoded_responses:
            if len(response.strip()) == 0:
                # Penalize empty responses
                final_score = -1.0
            else:
                base_score = self.reward_model.get_reward(response)
                word_count = len(response.split())
                length_bonus = min(word_count / 20.0, 1.0) * 0.5
                final_score = base_score + length_bonus
            
            rewards.append(final_score)
            print(f"Response: '{response}'\nReward: {final_score:.3f}\n")
        
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        
        # PPO step expects query_tensors and response_tensors separately
        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        return decoded_responses, rewards, stats