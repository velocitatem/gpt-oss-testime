from vllm import LLMEngine, EngineArgs, SamplingParams, TokensPrompt
import torch


class TokenGenerator:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        enable_output_sampling: bool = False,
        noise_std: float = 0.1,
        noise_spread: float = 0.2,
    ):
        args = EngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.engine = LLMEngine.from_engine_args(args)
        self.request_id = 0
        self.enable_output_sampling = enable_output_sampling
        self.noise_std = noise_std
        self.noise_spread = noise_spread

        # Apply sampling to model if enabled
        if self.enable_output_sampling:
            self._enable_model_sampling()

    def _enable_model_sampling(self):
        """Enable output space sampling in the VLLM model."""
        try:
            # Get the model from VLLM engine
            model = self.engine.model_executor.driver_worker.model_runner.model.model

            # Hook into transformer blocks to add sampling
            for block in model.layers:
                if hasattr(block, 'mlp'):
                    self._add_sampling_hook(block.mlp)
        except AttributeError as e:
            print(f"Warning: Could not enable sampling - {e}")

    def _add_sampling_hook(self, mlp_module):
        """Add forward hook to MLP module for output sampling."""
        original_forward = mlp_module.forward

        def sampling_forward(self_module, *args, **kwargs):
            # Get original output
            output = original_forward(*args, **kwargs)

            # Apply sampling if enabled and in training mode
            if self.enable_output_sampling and self_module.training:
                output = self._apply_output_sampling(output)

            return output

        # Replace forward method
        mlp_module.forward = sampling_forward.__get__(mlp_module, type(mlp_module))

    def _apply_output_sampling(self, mlp_output):
        """Apply normal distribution noise around random indices."""
        if not isinstance(mlp_output, torch.Tensor):
            return mlp_output

        batch_size, seq_len, hidden_size = mlp_output.shape

        # For each sample in batch, pick a random center index
        center_indices = torch.randint(0, hidden_size, (batch_size,), device=mlp_output.device)

        # Create noise tensor
        noise = torch.zeros_like(mlp_output)

        for i in range(batch_size):
            center_idx = center_indices[i].item()

            # Create position indices relative to center
            positions = torch.arange(hidden_size, device=mlp_output.device, dtype=torch.float32)
            distances = torch.abs(positions - center_idx) / hidden_size

            # Generate normal distribution weights (low-high spread around center)
            weights = torch.exp(-0.5 * (distances / self.noise_spread) ** 2)

            # Generate random noise and scale by weights
            random_noise = torch.randn(seq_len, hidden_size, device=mlp_output.device, dtype=mlp_output.dtype)
            noise[i] = random_noise * weights.unsqueeze(0) * self.noise_std

        return mlp_output + noise

    def set_output_sampling(self, enabled: bool):
        """Enable or disable output sampling for exploration."""
        self.enable_output_sampling = enabled
        try:
            model = self.engine.model_executor.driver_worker.model_runner.model.model
            for block in model.layers:
                if hasattr(block, 'mlp'):
                    block.mlp.training = enabled
        except AttributeError:
            pass

    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int] | None = None,
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False):
        if max_tokens == 0:
            max_tokens = None
        request_id = str(self.request_id)
        self.request_id += 1
        sampling_params = SamplingParams(temperature=temperature,
                                         max_tokens=max_tokens,
                                         stop_token_ids=stop_tokens,
                                         logprobs=0 if return_logprobs else None)
        prompt = TokensPrompt(prompt_token_ids=prompt_tokens)
        self.engine.add_request(request_id, prompt, sampling_params)
        last_token_id = []
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()

            # Check if we have any outputs
            if not step_outputs or len(step_outputs) == 0:
                continue

            first_output = step_outputs[0]
            if not hasattr(first_output, 'outputs') or len(first_output.outputs) == 0:
                continue

            output = first_output.outputs[0]
            token_ids = output.token_ids
            logprobs_list = output.logprobs if hasattr(output, "logprobs") else None
            new_token_ids = token_ids[len(last_token_id):]
            new_logprobs = logprobs_list[len(last_token_id):] if logprobs_list is not None else [None] * len(new_token_ids)

            for token_id, logprobs in zip(new_token_ids, new_logprobs):
                last_token_id.append(token_id)
                if return_logprobs:
                    logprob_val = None
                    if logprobs is not None and token_id in logprobs:
                        logprob_val = logprobs[token_id].logprob
                    yield (token_id, logprob_val)
                else:
                    yield token_id
                if stop_tokens is not None and token_id in stop_tokens:
                    break
