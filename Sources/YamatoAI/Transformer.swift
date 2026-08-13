import Foundation

struct Transformer {

    var blocks: [TransformerBlock]


    // =====================================
    // Forward
    // =====================================

    func calculate(

        embeddings: [EmbeddingVector]

    ) -> [EmbeddingVector] {

        var currentEmbeddings = embeddings


        for block in blocks {

            currentEmbeddings =
                block.calculate(

                    embeddings:
                        currentEmbeddings

                )
        }


        return currentEmbeddings
    }


    // =====================================
    // Update
    // =====================================

    mutating func update(

        gradient: TransformerGradientResult,

        learningRate: Float

    ) {

        // =====================================
        // Safety Check
        // =====================================

        guard blocks.count ==
                gradient.blockGradients.count else {

            return
        }


        // =====================================
        // Block Update
        // =====================================

        for index in blocks.indices {

            blocks[index].update(

                gradient:
                    gradient.blockGradients[index],

                learningRate:
                    learningRate

            )
        }
    }
}
