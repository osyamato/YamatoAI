import Foundation

struct OutputLayerGradient {

    func calculate(

        vector: EmbeddingVector,

        outputGradient: [Float]

    ) -> Matrix {

        var gradientValues: [[Float]] = []

        for inputValue in vector.values {

            var row: [Float] = []

            for gradient in outputGradient {

                let weightGradient = inputValue * gradient

                row.append(weightGradient)

            }

            gradientValues.append(row)
        }

        return Matrix(

            values: gradientValues

        )
    }
}
