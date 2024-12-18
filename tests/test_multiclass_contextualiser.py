import pytest
import torch

from neural_bandits.utils.multiclass import MultiClassContextualiser


class TestMultiClassContextualiser:
    @pytest.mark.parametrize("n_arms", [1, 2, 3, 5])
    def test_init(self, n_arms: int) -> None:
        # Test initialization works correctly
        contextualiser = MultiClassContextualiser(n_arms=n_arms)
        assert contextualiser.n_arms == n_arms, "n_arms is not correctly set."

    @pytest.mark.parametrize(
        "batch_size,n_features,n_arms", [(1, 3, 2), (2, 4, 3), (5, 10, 1), (4, 2, 5)]
    )
    def test_contextualise_shape(
        self, batch_size: int, n_features: int, n_arms: int
    ) -> None:
        # Given a certain input shape, test that the output shape is as expected
        contextualiser = MultiClassContextualiser(n_arms=n_arms)
        feature_vector = torch.randn(batch_size, n_features)

        output = contextualiser.contextualise(feature_vector)

        expected_shape = (batch_size, n_arms, n_features * n_arms)
        assert (
            output.shape == expected_shape
        ), f"Output shape {output.shape} does not match expected {expected_shape}"

    def test_contextualise_values(self) -> None:
        # Test against a known input and verify correctness of output values
        n_arms = 3
        feature_vector = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape: (2, 2)
        # Expected behavior:
        # kron(I_3, X) with X = [[1,2],[3,4]] should create a block diagonal structure:
        # For batch_size=2, n_features=2, n_arms=3, output shape is (2,3,6).
        #
        # Each batch element:
        # I_3 = [[1,0,0],[0,1,0],[0,0,1]]
        # kron(I_3, X):
        # = [[1*X, 0*X, 0*X],
        #    [0*X, 1*X, 0*X],
        #    [0*X, 0*X, 1*X]]
        #
        # For the first batch element (X = [1,2]):
        #    [ [1,2,0,0,0,0],
        #      [0,0,1,2,0,0],
        #      [0,0,0,0,1,2] ]
        #
        # For the second batch element (X = [3,4]):
        #    [ [3,4,0,0,0,0],
        #      [0,0,3,4,0,0],
        #      [0,0,0,0,3,4] ]

        contextualiser = MultiClassContextualiser(n_arms=n_arms)
        output = contextualiser.contextualise(feature_vector)

        assert output.shape == (2, 3, 6), "Output shape is incorrect."

        # Check first batch element
        expected_first = torch.tensor(
            [
                [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
            ]
        )
        # Check second batch element
        expected_second = torch.tensor(
            [
                [3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 3.0, 4.0],
            ]
        )

        print(output)

        assert torch.allclose(
            output[0], expected_first
        ), "First batch output is incorrect."
        assert torch.allclose(
            output[1], expected_second
        ), "Second batch output is incorrect."

    def test_gradient_propagation(self) -> None:
        # Ensure that gradients can flow back through the contextualiser
        n_arms = 2
        batch_size = 2
        n_features = 3
        contextualiser = MultiClassContextualiser(n_arms=n_arms)
        feature_vector = torch.randn(batch_size, n_features, requires_grad=True)

        output = contextualiser.contextualise(feature_vector)
        loss = output.sum()
        loss.backward()  # type: ignore

        assert (
            feature_vector.grad is not None
        ), "Gradients are not flowing back to the input."
        assert (
            feature_vector.grad.shape == feature_vector.shape
        ), "Gradient shape does not match feature_vector shape."