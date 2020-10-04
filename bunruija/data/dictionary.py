
class Dictionary:
    def __init__(self, pad='<pad>', eos='</s>', bos='<s>'):
        self.pad = pad
        self.eos = eos
        self.bos = bos
        self.elements = []
        self.count = []
        self.index_to_element = {}

    def __contains__(self, element):
        return element in self.index_to_element

    def add(self, element):
        if element not in self:
            self.elements.append(element)
            index = len(self.index_to_element)
            self.index_to_element[element] = index
            self.count.append(1)
        else:
            index = self.get_index(element)
            self.count[index] += 1
        return index

    def get_element(self, index):
        return self.elements[index]

    def get_index(self, element):
        return self.index_to_element[element]

    def __len__(self):
        return len(self.elements)
