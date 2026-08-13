import Foundation

struct TransformerBlockGradientResult {

    let inputGradients: [EmbeddingVector]

    let multiHeadAttentionGradient:
        MultiHeadAttentionGradientResult

    let w1Gradient: Matrix

    let w2Gradient: Matrix
}
