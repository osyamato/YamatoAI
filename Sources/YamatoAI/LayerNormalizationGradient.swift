import Foundation

struct LayerNormalizationGradient {

    func calculate(

        input: EmbeddingVector,

        outputGradient: EmbeddingVector,

        epsilon: Float = 0.00001

    ) -> EmbeddingVector {

        let values = input.values

        let gradients = outputGradient.values

        let count = values.count

        guard count > 0 else {

            return EmbeddingVector(
                values: []
            )
        }


        // -------------------------
        // Mean
        // -------------------------

        let mean =
            values.reduce(0, +)
            / Float(count)


        // -------------------------
        // Variance
        // -------------------------

        var variance: Float = 0

        for value in values {

            let difference =
                value - mean

            variance +=
                difference * difference
        }

        variance /=
            Float(count)


        // -------------------------
        // Inverse Standard Deviation
        // -------------------------

        let inverseStandardDeviation =
            1 / sqrt(
                variance + epsilon
            )


        // -------------------------
        // Gradient averages
        // -------------------------

        let gradientMean =
            gradients.reduce(0, +)
            / Float(count)


        var gradientNormalizedMean: Float = 0

        for index in values.indices {

            let normalizedValue =
                (values[index] - mean)
                * inverseStandardDeviation

            gradientNormalizedMean +=
                gradients[index]
                * normalizedValue
        }

        gradientNormalizedMean /=
            Float(count)


        // -------------------------
        // Input Gradient
        // -------------------------

        var inputGradients: [Float] = []

        for index in values.indices {

            let normalizedValue =
                (values[index] - mean)
                * inverseStandardDeviation

            let gradient =
                inverseStandardDeviation
                * (
                    gradients[index]
                    - gradientMean
                    - normalizedValue
                    * gradientNormalizedMean
                )

            inputGradients.append(
                gradient
            )
        }


        return EmbeddingVector(

            values: inputGradients

        )
    }
}
