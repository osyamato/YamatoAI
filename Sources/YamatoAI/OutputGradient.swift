import Foundation

struct OutputGradient {

    func calculate(

        probabilities: [Float],

        correctTokenID: Int

    ) -> [Float] {

        var gradients: [Float] = []

        for index in probabilities.indices {

            let probability = probabilities[index]

            if index == correctTokenID {

                let gradient = probability - 1.0

                gradients.append(gradient)

            }
            else {

                let gradient = probability

                gradients.append(gradient)

            }
            
        }
        
      

        return gradients
    }
}
