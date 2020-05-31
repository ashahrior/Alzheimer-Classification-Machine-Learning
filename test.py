import xlsxwriter as xl

excel_loc = r"E:\\THESIS\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\LogRegClassifier\\"
for i in range(1,6):
    outWorkbook = xl.Workbook(excel_loc+'out{}.xlsx'.format(i))
    outSheet = outWorkbook.add_worksheet()

    names = ['ashef','shahrior','ashfak']
    val = [313,342,124]
    age = [24,22,19]

    outSheet.write('A1','Names')
    outSheet.write('B1','Scores')
    outSheet.write('C1','Age')

    for item in range(len(names)):
        outSheet.write(item+1,0,names[item])
        outSheet.write(item+1,1,val[item])
        outSheet.write(item+1,2,age[item])
    outWorkbook.close()