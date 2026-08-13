import Foundation

struct ReLU {

    func calculate(

        vector: EmbeddingVector

    ) -> EmbeddingVector {

        var activatedValues: [Float] = []

        for value in vector.values {

            let activatedValue = max(

                0,

                value

            )

            activatedValues.append(activatedValue)

        }

        return EmbeddingVector(

            values: activatedValues

        )
    }
}
