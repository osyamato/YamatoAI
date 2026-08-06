import Foundation

struct Query {

    func make(

        from embedding: EmbeddingVector,

        using wq: Matrix

    ) -> EmbeddingVector {

        return wq.multiply(

            vector: embedding

        )

    }

}
