import Foundation

struct OutputLayer {

    var weight: Matrix

    func calculate(

        vector: EmbeddingVector

    ) -> EmbeddingVector {

        let output = weight.multiply(

            vector: vector

        )

        return output
    }

    mutating func update(

        gradient: Matrix,

        learningRate: Float

    ) {

        var newWeightValues: [[Float]] = []

        for rowIndex in weight.values.indices {

            var newRow: [Float] = []

            for columnIndex in weight.values[rowIndex].indices {

                let currentWeight = weight.values[rowIndex][columnIndex]

                let currentGradient = gradient.values[rowIndex][columnIndex]

                let newWeight = currentWeight - learningRate * currentGradient

                newRow.append(newWeight)

            }

            newWeightValues.append(newRow)
        }

        weight = Matrix(

            values: newWeightValues

        )
    }
}
