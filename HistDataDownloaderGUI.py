# -*- coding: utf-8 -*-

import ibapi
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from ibapi.wrapper import EWrapper
from ibapi.client import EClient

from threading import Thread

from time import sleep, strftime, time, localtime

class msgHandler(object):
    def handleHistoricalData(self, reqId: int, bar: ibapi.common.BarData):
        pass
    
    def handleHistoricalDataEnd(self, reqId:int, start:str, end:str):
        pass

class IBWrapper(EWrapper):
    def __init__(self):
        EWrapper.__init__(self)
        self.exporter = None
        self.contractDetailsObtained  = False
    
    def init_withExporter(self, exporter):
        EWrapper.__init__(self)
        self.exporter = exporter
        self.contractDetailsObtained  = False
        
    def historicalData(self, reqId: int, bar: ibapi.common.BarData):
        """ returns the requested historical data bars

        reqId - the request's identifier
        date  - the bar's date and time (either as a yyyymmss hh:mm:ssformatted
             string or as system time according to the request)
        open  - the bar's open point
        high  - the bar's high point
        low   - the bar's low point
        close - the bar's closing point
        volume - the bar's traded volume if available
        count - the number of trades during the bar's timespan (only available
            for TRADES).
        WAP -   the bar's Weighted Average Price
        hasGaps  -indicates if the data has gaps or not. """
        
        if self.exporter == None:
            self.logAnswer(ibapi.utils.current_fn_name(), vars())
            return
        else:
            self.exporter.handleHistoricalData(reqId, bar)
    
    def historicalDataEnd(self, reqId:int, start:str, end:str):
        if self.exporter == None:
            self.logAnswer(ibapi.utils.current_fn_name(), vars())
            return
        else:
            self.exporter.handleHistoricalDataEnd(reqId, start, end)
    
    def contractDetails(self, reqId:int, contractDetails:ibapi.contract.ContractDetails):
        self.resolved_contract = contractDetails.summary
        
    def contractDetailsEnd(self, reqId:int):
        self.contractDetailsObtained = True
        
class IBClient(EClient):
    
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)

class tApp(IBWrapper, IBClient):
    
    def __init__(self, IPAddress, PortId, ClientId, msgHandler):
        IBWrapper.init_withExporter(self, msgHandler)
        IBClient.__init__(self, wrapper = self)
        
        self.connect(IPAddress, PortId, ClientId)
        
        thread = Thread(target = self.run, name = "MainThread")
        thread.start()
        
        setattr(self, "_thread", thread)
        
class Application(tk.Frame, msgHandler):
    
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        msgHandler.__init__(self)
        
        self.port = 7496
        self.clientID = 25
        self.grid()
        self.create_widgets()
        self.isConnected = False
        
    def create_widgets(self):
        
        now = strftime('%Y%m%d %H:%M:%S', localtime(int(time())))
        myfont = ('Arial', 12)
        
        self.btnConnect = tk.ttk.Button(self, text = 'Connect', command = self.connect_to_tws)
        self.btnConnect.grid(row=1, column=1, sticky=tk.W)
        
        self.btnGetData = ttk.Button(self, text = 'GetData', command = self.getHistData)
        self.btnGetData.grid(row=1, column=2, sticky=tk.W)

        self.label_datetime = tk.Label(root, font=myfont, text = 'End Datetime').grid(row=3, column=0)
        self.label_duration = tk.Label(root, font=myfont, text = 'Duration').grid(row=3, column=1)
        self.label_barsize = tk.Label(root, font=myfont, text = 'BarSize').grid(row=3, column=2)
        
        varDateTimeEnd.set(now)
        
        self.cbDateTimeEnd = tk.Entry(root, font=myfont, textvariable = varDateTimeEnd).grid(row=4, column=0)
        self.cbDuration = ttk.Combobox(root, font = myfont, textvariable = varDuration)
        self.cbDuration['values'] = ('1 Y', '1 M', '6 M', '1 D', '7 D')
        self.cbDuration.grid(row=4, column=1)
        
        self.cbBarSize = ttk.Combobox(root, font=myfont, textvariable = varBarSize)
        self.cbBarSize['values'] = ('1 day', '1 min', '2 mins', '5 mins')
        self.cbBarSize.grid(row=4, column=2)
        
        
        
        self.listbox1 = tk.Listbox(root, font=("",12), width=75, height=30)
        self.listbox1.grid(row=6, column=0, columnspan=5, padx=5, pady=5, sticky='w')
        
        self.msgBox = tk.Listbox(root, font=("",12), width=75, height=10)
        self.msgBox.grid(row=7, column=0, columnspan=5, padx=5, pady=5, sticky='w')
        
    def connect_to_tws(self):
        
        if self.isConnected:
            self.tws_client.disconnect()
            self.btnConnect.config(text = 'Connect')
            self.msgBox.insert(tk.END, "Disconnected From IB")
            self.isConnected = False
        else:
            self.tws_client = tApp('LocalHost', self.port, self.clientID, self)
            timePassed = 0
            while not(self.tws_client.isConnected() ):
                sleep(0.1)
                timePassed += 0.1
                if (timePassed >5):
                    self.msgBox.insert(tk.END, "waited more than 5 secs to establish connection to TWS")
            
            self.isConnected = True
            self.msgBox.insert(tk.END, "Successfully connected to IB")
            self.btnConnect.config(text = "Disconnect")
            
    def getHistData(self):
        if not(self.isConnected):
            self.msgBox.insert(tk.END, "Not Connected to IB yet")
            return
        
        self.listbox1.delete(0, tk.END)
        
        self.contract = ibapi.contract.Contract()
        self.contract.symbol = "COIL"
        self.contract.secType = "FUT"
        self.contract.exchange = "IPE"
        self.contract.currency = "USD"
        self.contract.lastTradeDateOrContractMonth = "201801"
        
        self.tws_client.reqContractDetails(reqId = 2, contract=self.contract)
        self.tws_client.contractDetailsObtained = False
        
        timePassed = 0
        while not(self.tws_client.contractDetailsObtained):
            sleep(0.1)
            timePassed += 0.1
            if (timePassed>10):
                self.msgBox.insert(tk.END, "Waited more than 10 secs for contract details request")
        
        self.msgBox.insert(tk.END, "Successfully obtained contract details")
        aContract = self.tws_client.resolved_contract
        aContract.includeExpired = True

        now = varDateTimeEnd.get()
        duration = varDuration.get()
        barsize = varBarSize.get()
        
        self.tws_client.reqHistoricalData(reqId = 1,
                                          contract = aContract,
                                          endDateTime = now,
                                          durationStr = duration,
                                          barSizeSetting = barsize,
                                          whatToShow = 'TRADES',
                                          useRTH = 1,
                                          formatDate = 1,
                                          keepUpToDate = False,
                                          chartOptions = [])
                                          
    def disconnect(self):
        if self.isConnected:
            self.tws_client.disconnect()
            self.isConnected = False
        
    def handleHistoricalData(self, reqId: int, bar: ibapi.common.BarData):
        str_open = str(bar.open)
        str_high = str(bar.high)
        str_low = str(bar.low)
        str_close = str(bar.close)
        str_volume = str(bar.volume)
        
        histData = bar.date + "," + str_open + "," + str_high + "," + str_low + "," + str_close + "," + str_volume
        self.listbox1.insert(tk.END, histData)
    
    def handleHistoricalDataEnd(self, reqId:int, start:str, end:str):
        self.msgBox.insert(tk.END, "Finished downloading historical data")

root = tk.Tk()

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        app.disconnect()
        root.destroy()

root.title("Historical Data from IB Python API")
root.geometry("800x1200")
root.attributes("-topmost", True)
root.protocol("WM_DELETE_WINDOW", on_closing)

varDateTimeEnd = tk.StringVar()
varDuration = tk.StringVar(root, value = "1 M")
varBarSize = tk.StringVar(root, value = "1 day")

app = Application(root)

root.mainloop()