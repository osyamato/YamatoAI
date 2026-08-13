import Foundation

struct ResidualConnection {

    func calculate(

        original: EmbeddingVector,

        transformed: EmbeddingVector

    ) -> EmbeddingVector {

        var residualValues: [Float] = []

        for index in original.values.indices {

            let value =

                original.values[index]

                +

                transformed.values[index]

            residualValues.append(value)

        }

        return EmbeddingVector(

            values: residualValues

        )
    }
}
