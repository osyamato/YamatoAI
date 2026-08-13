import Foundation

struct FeedForward {

    var w1: Matrix

    var w2: Matrix

    func calculate(

        vector: EmbeddingVector

    ) -> EmbeddingVector {

        let hidden = w1.multiply(

            vector: vector

        )

        let activated = ReLU().calculate(

            vector: hidden

        )

        let output = w2.multiply(

            vector: activated

        )

        return output
    }

    mutating func update(

        w1Gradient: Matrix,

        w2Gradient: Matrix,

        learningRate: Float

    ) {

        // MARK: - W1 Update

        var newW1Values: [[Float]] = []

        for rowIndex in w1.values.indices {

            var newRow: [Float] = []

            for columnIndex in w1.values[rowIndex].indices {

                let currentWeight =
                    w1.values[rowIndex][columnIndex]

                let currentGradient =
                    w1Gradient.values[rowIndex][columnIndex]

                let newWeight =
                    currentWeight
                    - learningRate * currentGradient

                newRow.append(newWeight)
            }

            newW1Values.append(newRow)
        }

        w1 = Matrix(

            values: newW1Values

        )


        // MARK: - W2 Update

        var newW2Values: [[Float]] = []

        for rowIndex in w2.values.indices {

            var newRow: [Float] = []

            for columnIndex in w2.values[rowIndex].indices {

                let currentWeight =
                    w2.values[rowIndex][columnIndex]

                let currentGradient =
                    w2Gradient.values[rowIndex][columnIndex]

                let newWeight =
                    currentWeight
                    - learningRate * currentGradient

                newRow.append(newWeight)
            }

            newW2Values.append(newRow)
        }

        w2 = Matrix(

            values: newW2Values

        )
    }
}
