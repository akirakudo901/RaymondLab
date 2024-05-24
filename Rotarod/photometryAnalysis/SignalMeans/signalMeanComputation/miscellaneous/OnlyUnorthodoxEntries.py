
import csv
import os

READ_PATH = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\Rotarod\photometryAnalysis\SignalMeans\signalMeanComputation\results\Q175_NumberOfPtABAndNoteEntries.csv"

CSV_FOLDER = os.path.dirname(READ_PATH)
CSV_NAME = "Q175_UnexpectedPtABorNoteEntries.csv"
CSV_PATH = os.path.join(CSV_FOLDER, CSV_NAME)

NOTE_ONSET_ONE_IDX = 1; NOTE_ONSET_TWO_IDX = 2
PTAB_ONSET_ONE_IDX = 3; PTAB_ONSET_TWO_IDX = 4

NOTE_ONSET_EXPECTED = ('2', '1')
PTAB_ONSET_EXPECTED = ('2', '1')

unexpected_entries = []

fields = []

with open(READ_PATH, mode ='r') as file:
    csvFile = csv.reader(file)
    fields = next(csvFile)
    for lines in csvFile:
        note_onset = (lines[NOTE_ONSET_ONE_IDX], 
                      lines[NOTE_ONSET_TWO_IDX])
        ptab_onset = (lines[PTAB_ONSET_ONE_IDX], 
                      lines[PTAB_ONSET_TWO_IDX])
        if note_onset != NOTE_ONSET_EXPECTED or \
           ptab_onset != PTAB_ONSET_EXPECTED:
            unexpected_entries.append(lines)

print("\n".join([",".join(inner_list) for inner_list in unexpected_entries]))

# writing to csv file
with open(CSV_PATH, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)
        
    # writing the fields  
    csvwriter.writerow(fields)
        
    # writing the data rows  
    csvwriter.writerows(unexpected_entries)