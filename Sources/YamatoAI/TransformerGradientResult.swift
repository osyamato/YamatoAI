import Foundation

struct TransformerGradientResult {

    // Transformer全体を逆伝播して

    // Embedding側へ戻るGradient

    let inputGradients: [EmbeddingVector]

    // 各TransformerBlockのGradient

    let blockGradients: [TransformerBlockGradientResult]

}
