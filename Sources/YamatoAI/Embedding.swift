import Foundation

struct Embedding {

    let dimension: Int

    private(set) var vectors: [EmbeddingVector]

    init(
        dimension: Int,
        vocabularySize: Int
    ) {

        self.dimension = dimension

        self.vectors = []

        for _ in 0..<vocabularySize {

            vectors.append(
                EmbeddingVector.random(
                    dimension: dimension
                )
            )

        }

    }

}
