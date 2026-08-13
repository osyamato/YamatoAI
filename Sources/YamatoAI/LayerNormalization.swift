import Foundation

struct LayerNormalization {

    func calculate(

        vector: EmbeddingVector

    ) -> EmbeddingVector {

        var total: Float = 0

        for value in vector.values {

            total += value

        }

        let mean = total / Float(vector.values.count)

        var differences: [Float] = []

        for value in vector.values {

            let difference = value - mean

            differences.append(difference)

        }

        var squaredDifferenceTotal: Float = 0

        for difference in differences {

            squaredDifferenceTotal += difference * difference

        }

        let variance =

            squaredDifferenceTotal

            /

            Float(differences.count)

        let epsilon: Float = 0.00001

        let standardDeviation = sqrt(

            variance + epsilon

        )

        var normalizedValues: [Float] = []

        for difference in differences {

            let normalizedValue =

                difference

                /

                standardDeviation

            normalizedValues.append(normalizedValue)

        }

        return EmbeddingVector(

            values: normalizedValues

        )
    }
}
