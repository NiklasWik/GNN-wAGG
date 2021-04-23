import smtplib
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders
import glob
import os 
from shutil import copyfile


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
    print("Sending results")
    smtp.sendmail(username, send_to, msg.as_string())
    smtp.quit()
  
def mail_GNNs(send_to, directory, note, password='', send_accs=False, send_all=False, seed=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path+'/'+directory+'results/mailresults.txt'
    path2 = dir_path+'/'+directory+'results/'
    f = open(path, "r")
    res = []
    keys = []
    for x in f:
        res.append(x.split(": ",1)[1].replace('\n',''))
        keys.append(x.split(": ",1)[0])
    dictt = dict(zip(keys, res))
    f.close()
    sub = dictt["model"]+", "+dictt["seed"]+", "+dictt["aggr_func"]+", "+dictt["dataset"]+", "+dictt["date"]
    msg = """
    {}, {}, seed {}, {}, params: {}
    test_acc: {}
    train_acc: {}
    epochs: {}
    avg_time_per_epoch (s): {}
    total_time (h): {}
    note: {}
    """.format(dictt["model"], dictt["aggr_func"], dictt["seed"], dictt["dataset"], dictt["params"], dictt["testacc"], dictt["trainacc"], dictt["epochs"], dictt["avg_time_per_epoch"], dictt["total_time"], note)
    
    if send_all == True:
        msg = """
        All files from out/{}/results appended
        note: {}
        """.format(dictt["model"], note)
        sub = "SUMMARY: "+dictt["model"]+", "+dictt["aggr_func"]+", "+dictt["dataset"]+", "+dictt["date"]
        files = glob.glob(path2 + '*.txt')
    else:
        files = []
        files.append(path)

    if seed is not None:
        copyfile(path, path2 + 'mailresults' + str(seed) + '.txt')

    if send_accs == True:
        files.append('accs.mat')
    send_mail(send_to, sub, msg, files, 'gnns-wagg')