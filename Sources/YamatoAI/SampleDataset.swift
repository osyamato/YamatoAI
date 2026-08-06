import Foundation

enum SampleDataset {

    static func make() -> Dataset {

        var dataset = Dataset()

        dataset.add(
            TrainingExample(
                input: "こんにちは",
                output: "やあ！",
                category: .greeting,
                style: .gentle
            )
        )

        dataset.add(
            TrainingExample(
                input: "疲れた",
                output: "少し休もう。",
                category: .emotion,
                style: .gentle
            )
        )

        return dataset
    }
}
