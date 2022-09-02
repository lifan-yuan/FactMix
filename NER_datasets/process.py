# with open('/home/yuanlifan/NER/NER_datasets/tech_news/test.txt') as f:
#     lines = f.readlines()

#     newlines = []

#     for line in lines:
#         line = line.strip().split(' ')
#         if line[0] == '':
#             newlines.append('\n')
#             continue

#         if line[1].startswith('S-'):
#             temp = 'B-'+line[1].split('-')[1]
#             newlines.append([line[0], temp])
#             continue
#         if line[1].startswith('E-'):
#             temp = 'I-'+line[1].split('-')[1]
#             newlines.append([line[0], temp])
#             continue
#         newlines.append(line)
#     f.close()

# with open('/home/yuanlifan/NER/NER_datasets/tech_news/test.txt', 'w') as f:
#     for line in newlines:
#         try:
#             print(line[0], line[1], file=f)
#         except:
#             print(file=f)
#     f.close()

for domain in ['ai', 'literature', 'music', 'politics', 'science']:

    with open(f'/home/lifan/NER/NER_datasets/{domain}/test.txt') as f:
        lines = f.readlines()
        newlines = []

        for line in lines:
            line = line.strip().split('\t')
            if line[0] == '':
                newlines.append('\n')
                continue

            if 'person' in line[1]:
                line[1] = line[1].split('-')[0] + '-' + 'PER'
            elif 'location' in line[1]:
                line[1] = line[1].split('-')[0] + '-' +  'LOC'
            elif 'organisation' in line[1]:
                line[1] = line[1].split('-')[0] + '-' +  'ORG'
            elif 'misc' in line[1]:
                line[1] = line[1].split('-')[0] + '-' +  'MISC'
            else:
                line[1] = 'O'

            newlines.append(line)
        f.close()

    with open(f'/home/lifan/NER/NER_datasets/{domain}/test.txt', 'w') as f:
        for line in newlines:
            try:
                print(line[0], line[1].upper(), file=f)
            except:
                print(file=f)
        f.close()

