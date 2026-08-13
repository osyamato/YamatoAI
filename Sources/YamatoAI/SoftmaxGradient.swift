import Foundation

struct SoftmaxGradient {

    func calculate(

        probabilities: [Float],

        outputGradient: [Float]

    ) -> [Float] {

        var scoreGradients: [Float] = []

        for i in probabilities.indices {

            var gradientSum: Float = 0

            for j in probabilities.indices {

                let softmaxDerivative: Float

                if i == j {

                    softmaxDerivative =
                        probabilities[i]
                        * (1 - probabilities[i])

                } else {

                    softmaxDerivative =
                        -probabilities[i]
                        * probabilities[j]
                }

                gradientSum +=
                    outputGradient[j]
                    * softmaxDerivative
            }

            scoreGradients.append(

                gradientSum

            )
        }

        return scoreGradients
    }
}
