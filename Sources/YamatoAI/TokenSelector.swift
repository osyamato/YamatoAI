import Foundation

struct TokenSelector {

    func select(

        probabilities: [Float]

    ) -> Int {

        var highestProbability = probabilities[0]

        var selectedIndex = 0

        for index in probabilities.indices {

            let probability = probabilities[index]

            if probability > highestProbability {

                highestProbability = probability

                selectedIndex = index
            }
        }

        return selectedIndex
    }
}
