import Foundation

struct PositionalEncoding {

    func calculate(

        embeddings: [EmbeddingVector]

    ) -> [EmbeddingVector] {

        var results: [EmbeddingVector] = []

        for position in embeddings.indices {

            let embedding = embeddings[position]

            var encodedValues: [Float] = []

            for dimensionIndex in embedding.values.indices {

                let dimension = Float(

                    embedding.values.count

                )

                let pairIndex = Float(

                    dimensionIndex / 2

                )

                let exponent =

                    (2.0 * pairIndex)

                    /

                    dimension

                let denominator = pow(

                    10000.0,

                    exponent

                )

                let angle =

                    Float(position)

                    /

                    denominator

                let positionalValue: Float

                if dimensionIndex % 2 == 0 {

                    positionalValue = sin(angle)

                } else {

                    positionalValue = cos(angle)

                }

                let encodedValue =

                    embedding.values[dimensionIndex]

                    +

                    positionalValue

                encodedValues.append(encodedValue)
            }

            let encodedEmbedding = EmbeddingVector(

                values: encodedValues

            )

            results.append(encodedEmbedding)
        }

        return results
    }
}
