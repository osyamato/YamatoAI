import Foundation

struct TransformerGradient {

    func calculate(

        embeddings: [EmbeddingVector],

        outputGradients: [EmbeddingVector],

        transformer: Transformer

    ) -> TransformerGradientResult {

        // =====================================
        // Forward時の各Block入力を保存
        // =====================================

        var blockInputs: [[EmbeddingVector]] = []

        var currentEmbeddings = embeddings


        for block in transformer.blocks {

            // このBlockに入った値を保存
            blockInputs.append(

                currentEmbeddings

            )


            // Forward
            currentEmbeddings =
                block.calculate(

                    embeddings: currentEmbeddings

                )
        }


        // =====================================
        // Backward
        // =====================================

        var currentGradients =
            outputGradients


        // BlockごとのGradientを
        // 一時的に逆順で保存
        var reversedBlockGradients:
            [TransformerBlockGradientResult] = []


        // =====================================
        // 最後のBlockから逆伝播
        // =====================================

        for blockIndex
            in transformer.blocks.indices.reversed() {

            let block =
                transformer.blocks[blockIndex]


            let blockGradient =
                TransformerBlockGradient().calculate(

                    embeddings:
                        blockInputs[blockIndex],

                    outputGradients:
                        currentGradients,

                    multiHeadAttention:
                        block.multiHeadAttention,

                    feedForward:
                        block.feedForward

                )


            // -------------------------
            // このBlockのGradientを保存
            // -------------------------

            reversedBlockGradients.append(

                blockGradient

            )


            // -------------------------
            // 一つ前のBlockへ戻すGradient
            // -------------------------

            currentGradients =
                blockGradient.inputGradients
        }


        // =====================================
        // Gradient順序を元に戻す
        //
        // Backwardでは
        //
        // Block 2
        // Block 1
        // Block 0
        //
        // の順で作ったので、
        //
        // Block 0
        // Block 1
        // Block 2
        //
        // に戻す
        // =====================================

        let blockGradients =
            Array(

                reversedBlockGradients.reversed()

            )


        // =====================================
        // Result
        // =====================================

        return TransformerGradientResult(

            inputGradients:
                currentGradients,

            blockGradients:
                blockGradients

        )
    }
}
