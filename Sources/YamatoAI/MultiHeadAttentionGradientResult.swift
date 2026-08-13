import Foundation

struct MultiHeadAttentionGradientResult {

    // MultiHeadAttentionの入力

    // embeddingsへ戻るGradient

    let inputGradients: [EmbeddingVector]

    // Output Projection

    // WoのGradient

    let woGradient: Matrix

    // 各Attention Headの

    // Wq / Wk / Wv Gradient

    let headGradients: [AttentionWeightGradientResult]

}
