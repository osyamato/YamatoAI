import Foundation

struct CrossEntropyLoss {

    func calculate(

        probabilities: [Float],

        correctTokenID: Int

    ) -> Float {

        let correctProbability = probabilities[correctTokenID]

        let loss = -log(correctProbability)

        return loss
    }
}
