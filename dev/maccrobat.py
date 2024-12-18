import tarfile
from io import BytesIO

from bratlib.data import BratFile, Entity
from datasets import Dataset, load_dataset


# Function to convert the dataset to BRAT format
def convert_to_brat_format(data: Dataset, output_file: str):
    with tarfile.open(output_file, 'w:gz') as tar:
        for idx, record in enumerate(data):
            # Write the text content to a virtual file and add to the archive
            txt_file = record['full_text'].encode('utf-8')
            tarinfo_txt = tarfile.TarInfo(name=f'{idx}.txt')
            tarinfo_txt.size = len(txt_file)
            tar.addfile(tarinfo_txt, BytesIO(txt_file))

            # Write the annotations to a virtual file and add to the archive
            annotations = BratFile.from_data(
                entities=[
                    Entity(
                        tag=entity['label'],
                        spans=[(entity['start'], entity['end'])],
                        mention=entity['text'],
                    )
                    for entity in record['ner_info']
                ]
            )
            ann_file = str(annotations).encode('utf-8')
            tarinfo_ann = tarfile.TarInfo(name=f'{idx}.ann')
            tarinfo_ann.size = len(ann_file)
            tar.addfile(tarinfo_ann, BytesIO(ann_file))


if __name__ == "__main__":
    with load_dataset("singh-aditya/MACCROBAT_biomedical_ner") as dataset:
        convert_to_brat_format(dataset['train'], f"../{dataset['train'].info.dataset_name}.tar.gz")
