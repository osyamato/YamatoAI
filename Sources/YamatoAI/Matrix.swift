import Foundation

struct Matrix {

    let values: [[Float]]

    func multiply(

        vector: EmbeddingVector

    ) -> EmbeddingVector {

        var matrixProduct: [Float] = []

        for columnIndex in values[0].indices {

            var sum: Float = 0

            for rowIndex in vector.values.indices {

                sum +=

                    vector.values[rowIndex]

                    *

                    values[rowIndex][columnIndex]

            }

            matrixProduct.append(sum)

        }

        return EmbeddingVector(

            values: matrixProduct

        )

    }

}
