import Foundation

struct FeedForwardWeightGradient {

    func calculate(

        input: EmbeddingVector,

        outputGradient: EmbeddingVector,

        w1: Matrix,

        w2: Matrix

    ) -> (

        w1Gradient: Matrix,

        w2Gradient: Matrix

    ) {

        // W1を通した値
        let hidden = w1.multiply(

            vector: input

        )

        // ReLU
        var activatedValues: [Float] = []

        for value in hidden.values {

            if value > 0 {

                activatedValues.append(value)

            } else {

                activatedValues.append(0)

            }
        }

        let activated = EmbeddingVector(

            values: activatedValues

        )

        // W2自身のGradient
        let w2Gradient = OutputLayerGradient().calculate(

            vector: activated,

            outputGradient: outputGradient.values

        )

        // W2を逆向きに通す
        let hiddenGradient = InputGradient().calculate(

            outputGradient: outputGradient.values,

            weight: w2

        )

        // ReLUを逆向きに通す
        var reluGradientValues: [Float] = []

        for index in hidden.values.indices {

            if hidden.values[index] > 0 {

                reluGradientValues.append(

                    hiddenGradient.values[index]

                )

            } else {

                reluGradientValues.append(0)

            }
        }

        // W1自身のGradient
        let w1Gradient = OutputLayerGradient().calculate(

            vector: input,

            outputGradient: reluGradientValues

        )

        return (

            w1Gradient: w1Gradient,

            w2Gradient: w2Gradient

        )
    }
}
