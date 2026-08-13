import Foundation

struct FeedForwardGradient {

    func calculate(

        input: EmbeddingVector,

        outputGradient: EmbeddingVector,

        w1: Matrix,

        w2: Matrix

    ) -> EmbeddingVector {

        let hidden = w1.multiply(

            vector: input

        )

        let hiddenGradient = InputGradient().calculate(

            outputGradient: outputGradient.values,

            weight: w2

        )

        var reluGradientValues: [Float] = []

        for index in hidden.values.indices {

            if hidden.values[index] > 0 {

                reluGradientValues.append(

                    hiddenGradient.values[index]

                )

            } else {

                reluGradientValues.append(

                    0

                )
            }
        }

        let inputGradient = InputGradient().calculate(

            outputGradient: reluGradientValues,

            weight: w1

        )

        return inputGradient
    }
}
