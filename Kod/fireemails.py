import smtplib
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders
import glob
import os 


def send_mail(send_to, subject, message, files=[], password=''):
    """Compose and send email with provided info and attachments.
    Args:
        send_to (list[str]): to name(s)
        subject (str): message title
        message (str): message body
        files (list[str]): list of file paths to be attached to email
        password (str): server auth password
    """
    username = 'mortssaerdna2@gmail.com'
    server = 'smtp.gmail.com'
    port = 587
    msg = MIMEMultipart()
    msg['From'] = username
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(message))
    print("yes")
    for path in files:
        part = MIMEBase('application', "octet-stream")
        with open(path, 'rb') as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment; filename="{}"'.format(Path(path).name))
        msg.attach(part)

    smtp = smtplib.SMTP(server, port)
    smtp.starttls()
    smtp.login(username, password)
    smtp.sendmail(username, send_to, msg.as_string())
    smtp.quit()
  
def mail_GNNs(send_to, directory, note, password='', send_accs=False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)

    f = open(dir_path+ directory + 'results/mailresults.txt', "r")
    res = []
    keys = []
    for x in f:
        res.append(x.split(": ",1)[1].replace('\n',''))
        keys.append(x.split(": ",1)[0])
    dictt = dict(zip(keys, res))
    print(dictt)
    f.close()
    print(dictt)
    sub = dictt["model"]+", "+dictt["seed"]+", "+dictt["aggr_func"]+", "+dictt["dataset"]+", "+dictt["date"]
    msg = """
    model: {}, aggr: {}
    seed: {}, dataset: {}, params: {}
    test_acc: {}, train_acc: {}
    epochs: {}, avg_time_per_epoch (s): {}
    total_time (h): {}
    note: {}""".format(dictt["model"], dictt["aggr_func"], dictt["seed"], dictt["dataset"], dictt["params"], dictt["testacc"], dictt["trainacc"], dictt["epochs"], dictt["avg_time_per_epoch"], dictt["total_time"], note)
    print(dictt)
    files = glob.glob(directory + 'results/mailresults.txt')
    print(dictt)
    if send_accs == True:
        files.append('accs.mat')
    send_mail(send_to, sub, msg, files, 'gnns-wagg')
    print(dictt)

note = "testing"
dire = '/out/GatedTest/'
#f = open(dir+'results/mailresults.txt')
#/content/GNN-wAGG/results/mailresults.txt
send_to = "andreascstrom@gmail.com"
mail_GNNs(send_to, dire, note, 'gnns-wagg')