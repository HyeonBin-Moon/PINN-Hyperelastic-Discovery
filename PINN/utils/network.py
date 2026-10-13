"""Neural network modules: the displacement PINN and the constitutive coefficients."""
import torch
import torch.nn as nn


class PINN(nn.Module):
    """Fully connected network (x, y, t) -> (u_x, u_y, u_z).

    Seven hidden SiLU layers of `width` units with Xavier-uniform weights and zero
    biases. Inputs are scaled as X / 100 and t / 10. u_z is not learned: it is the
    prescribed out-of-plane displacement t / 10 * f33.

    Layer names, creation order and the (unused) `log_sigma_x/y` parameters are
    kept as in the original script, so random initialization and saved
    checkpoints are unchanged.
    """

    def __init__(self, width=32, f33=-0.5):
        super().__init__()
        n = width
        self.f33 = f33
        self.input_layer = nn.Linear(3, n)
        self.hidden_layer_1 = nn.Linear(n, n)
        self.hidden_layer_2 = nn.Linear(n, n)
        self.hidden_layer_3 = nn.Linear(n, n)
        self.hidden_layer_4 = nn.Linear(n, n)
        self.hidden_layer_5 = nn.Linear(n, n)
        self.hidden_layer_6 = nn.Linear(n, n)
        self.output_layer = nn.Linear(n, 2)
        self.activation = nn.SiLU()

        # Unused, but part of model.parameters() and the checkpoint.
        self.log_sigma_x = nn.Parameter(torch.zeros(5))
        self.log_sigma_y = nn.Parameter(torch.zeros(5))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, X, time):
        """X: (n, 2) reference coordinates, time: (n, 1) increment index 1..10."""
        time_scaler = time.clone().view(-1)
        X = (X - 0) / 100
        time = (time - 0) / 10
        time = time.view(-1, 1)
        x_input = torch.cat([X, time], dim=1)  # (n, 3)

        x = self.activation(self.input_layer(x_input))
        x = self.activation(self.hidden_layer_1(x))
        x = self.activation(self.hidden_layer_2(x))
        x = self.activation(self.hidden_layer_3(x))
        x = self.activation(self.hidden_layer_4(x))
        x = self.activation(self.hidden_layer_5(x))
        x = self.activation(self.hidden_layer_6(x))
        x = self.output_layer(x)

        output = torch.zeros(x.size(0), 3, dtype=torch.float32, device=x.device)
        output[:, 0] = x[:, 0] * 1
        output[:, 1] = x[:, 1] * 1
        output[:, 2] = time_scaler / 10 * self.f33
        return output


class ConstitutiveParams(nn.Module):
    """The library coefficients theta (initialized to zero) and their history."""

    def __init__(self, device, num_terms=57):
        super().__init__()
        # torch.rand(...) * 0 is kept: it consumes the RNG exactly as the original script.
        self.raw_params = nn.Parameter(torch.rand(num_terms, device=device, requires_grad=True) * (0.0))
        self.param_history = []

    @property
    def params(self):
        return self.raw_params

    def update_constitutive_params(self, new_params):
        with torch.no_grad():
            self.raw_params.copy_(new_params)

    def save_param_history(self):
        self.param_history.append(self.params.detach().cpu().numpy())
