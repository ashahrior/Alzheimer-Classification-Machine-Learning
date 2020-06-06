for i in range(len(headers)-3):
                    sheet.write(line+1, i, combo_list[c][i])

                sheet.write(line+1, (len(headers)-1), best_score*100)
                sheet.write(line+1, (len(headers)-2), serial)
                sheet.write(line+1, (len(headers)-3), best_score)