import Foundation

struct AttentionWeightGradientResult {

    let inputGradients: [EmbeddingVector]

    let wqGradient: Matrix

    let wkGradient: Matrix

    let wvGradient: Matrix
}
