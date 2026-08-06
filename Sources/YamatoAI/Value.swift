import Foundation

struct Value {

    func make(

        from embedding: EmbeddingVector,

        using wv: Matrix

    ) -> EmbeddingVector {

        return wv.multiply(

            vector: embedding

        )

    }

}

