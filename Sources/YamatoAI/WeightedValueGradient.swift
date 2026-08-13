import Foundation

struct WeightedValueGradient {

    func calculate(

        probabilities: [Float],

        values: [EmbeddingVector],

        outputGradient: EmbeddingVector

    ) -> (

        probabilityGradients: [Float],

        valueGradients: [EmbeddingVector]

    ) {

        var probabilityGradients: [Float] = []

        var valueGradients: [EmbeddingVector] = []


        // Probability側のGradient
        for value in values {

            let gradient = DotProduct().calculate(

                between: value,

                and: outputGradient

            )

            probabilityGradients.append(

                gradient

            )
        }


        // Value側のGradient
        for probability in probabilities {

            var gradientValues: [Float] = []

            for gradient in outputGradient.values {

                let valueGradient =
                    probability * gradient

                gradientValues.append(

                    valueGradient

                )
            }

            valueGradients.append(

                EmbeddingVector(

                    values: gradientValues

                )
            )
        }


        return (

            probabilityGradients,

            valueGradients

        )
    }
}
