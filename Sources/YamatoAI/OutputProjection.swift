import Foundation

struct OutputProjection {

    func calculate(

        vector: EmbeddingVector,

        using wo: Matrix

    ) -> EmbeddingVector {

        return wo.multiply(

            vector: vector

        )

    }

}
