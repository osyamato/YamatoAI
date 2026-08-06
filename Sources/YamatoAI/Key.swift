import Foundation

struct Key {

    func make(

        from embedding: EmbeddingVector,

        using wk: Matrix

    ) -> EmbeddingVector {

        return wk.multiply(

            vector: embedding

        )

    }

}
