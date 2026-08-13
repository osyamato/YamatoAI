import Foundation

struct ResidualGradient {

    func calculate(

        outputGradient: EmbeddingVector

    ) -> ResidualGradientResult {

        let inputGradient =
            outputGradient

        let branchGradient =
            outputGradient

        return ResidualGradientResult(

            inputGradient: inputGradient,

            branchGradient: branchGradient

        )
    }
}
