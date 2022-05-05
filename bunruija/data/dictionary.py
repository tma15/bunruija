class Dictionary:
    def __init__(self, pad="<pad>", eos="</s>", bos="<s>"):
        self.elements = []
        self.count = []
        self.index_to_element = {}

        self.pad = pad
        self.eos = eos
        self.bos = bos

        self.add(self.pad)
        self.add(self.eos)
        self.add(self.bos)

    def __contains__(self, element):
        return element in self.index_to_element

    def add(self, element, n=1):
        if element not in self:
            self.elements.append(element)
            index = len(self.index_to_element)
            self.index_to_element[element] = index
            self.count.append(n)
        else:
            index = self.get_index(element)
            self.count[index] += n
        return index

    def get_element(self, index):
        return self.elements[index]

    def get_index(self, element):
        return self.index_to_element[element]

    def __len__(self):
        return len(self.elements)
