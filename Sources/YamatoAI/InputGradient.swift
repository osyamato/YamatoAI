import Foundation

struct InputGradient {

    func calculate(

        outputGradient: [Float],

        weight: Matrix

    ) -> EmbeddingVector {

        var inputGradients: [Float] = []

        for rowIndex in weight.values.indices {

            var gradientSum: Float = 0

            for columnIndex in outputGradient.indices {

                let gradient = weight.values[rowIndex][columnIndex]
                    * outputGradient[columnIndex]

                gradientSum += gradient
            }

            inputGradients.append(gradientSum)
        }

        return EmbeddingVector(

            values: inputGradients

        )
    }
}
