import Foundation

struct Dataset {

    private(set) var examples: [TrainingExample] = []

    mutating func add(_ example: TrainingExample) {

        examples.append(example)

    }

}



